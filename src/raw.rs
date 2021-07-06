use std::cmp;
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::mem;
use std::num::NonZeroUsize;
use std::ops::{Index, IndexMut};
use std::ptr::NonNull;

#[repr(C)]
struct Header<T> {
    _len: usize,
    _cap: NonZeroUsize,
    _boo: PhantomData<T>,
}

impl<T> Header<T> {
    pub fn allocate(cap: NonZeroUsize) -> NonNull<Header<T>> {
        Self::header_with_capacity(cap)
    }

    pub unsafe fn deallocate(head: *mut Header<T>) {
        let cap = (&*head).capacity();
        std::alloc::dealloc(head as *mut u8, Self::layout(cap))
    }

    fn len(&self) -> usize {
        self._len
    }

    fn set_len(&mut self, len: usize) {
        self._len = len
    }

    fn capacity(&self) -> usize {
        self._cap.get()
    }

    fn set_capacity(&mut self, cap: NonZeroUsize) {
        self._cap = cap;
    }

    fn data(&self) -> *mut T {
        let padding = Self::padding();
        let header_size = mem::size_of::<Self>();
        let ptr = self as *const Self as *mut Self as *mut u8;
        let offset = header_size + padding;
        unsafe { ptr.offset(offset as isize) as *mut T }
    }

    fn alloc_align() -> usize {
        cmp::max(mem::align_of::<T>(), mem::align_of::<Self>())
    }

    fn layout(cap: usize) -> std::alloc::Layout {
        unsafe {
            std::alloc::Layout::from_size_align_unchecked(
                Self::alloc_size(cap),
                Self::alloc_align(),
            )
        }
    }

    fn padding() -> usize {
        let alloc_align = Self::alloc_align();
        let header_size = mem::size_of::<Self>();
        alloc_align.saturating_sub(header_size)
    }

    fn alloc_size(cap: usize) -> usize {
        // Compute "real" header size with pointer math
        let header_size = mem::size_of::<Self>();
        let elem_size = mem::size_of::<T>();
        let padding = Self::padding();
        let data_size = match elem_size.checked_mul(cap) {
            Some(data_size) if data_size <= isize::MAX as usize => data_size,
            _ => panic!("capacity overflow"),
        };
        let full_size = match data_size.checked_add(header_size + padding) {
            Some(full_size) if full_size <= isize::MAX as usize => full_size,
            _ => panic!("capacity overflow"),
        };
        full_size
    }

    fn header_with_capacity(cap: NonZeroUsize) -> NonNull<Header<T>> {
        unsafe {
            let layout = Self::layout(cap.get());
            let header = std::alloc::alloc(layout) as *mut Header<T>;
            if header.is_null() {
                std::alloc::handle_alloc_error(layout)
            }
            (*header).set_capacity(cap);
            (*header).set_len(0);
            NonNull::new_unchecked(header)
        }
    }
}

#[repr(transparent)]
struct Segment<T>(Option<NonNull<Header<T>>>);

impl<T> Segment<T> {
    fn new(cap: NonZeroUsize) -> Self {
        Segment(Some(Header::<T>::allocate(cap)))
    }

    fn len(&self) -> usize {
        match self.0 {
            Some(ptr) => unsafe { ptr.as_ref() }.len(),
            None => 0,
        }
    }

    fn capacity(&self) -> usize {
        match self.0 {
            Some(ptr) => unsafe { ptr.as_ref() }.capacity(),
            None => 0,
        }
    }

    fn get(&self, index: usize) -> Option<&T> {
        match self.0 {
            Some(ptr) => unsafe {
                let header = ptr.as_ref();
                match header.len() {
                    l if l > index => header.data().offset(index as isize).as_ref(),
                    _ => None,
                }
            },
            None => None,
        }
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        match self.0 {
            Some(ptr) => unsafe {
                let header = ptr.as_ref();
                match header.len() {
                    l if l > index => header.data().offset(index as isize).as_mut(),
                    _ => None,
                }
            },
            None => None,
        }
    }

    fn push(&mut self, val: T) {
        match self.0 {
            Some(mut ptr) => {
                let header = unsafe { ptr.as_mut() };
                let len = header.len();
                let cap = header.capacity();
                assert!(len < cap, "segment overflow");
                let ptr = header.data();
                let elt_ptr = unsafe { ptr.offset(len as isize) };
                unsafe { std::ptr::write(elt_ptr, val) }
                header.set_len(len + 1);
            }
            None => unreachable!("use of dropped segment"),
        }
    }

    fn pop(&mut self) -> Option<T> {
        match self.0 {
            Some(mut ptr) => {
                let header = unsafe { ptr.as_mut() };
                let len = header.len();
                match len {
                    0 => None,
                    mut len => {
                        len -= 1;
                        header.set_len(len);
                        let ptr = header.data();
                        let elt_ptr = unsafe { ptr.offset(len as isize) };
                        Some(unsafe { std::ptr::read(elt_ptr) })
                    }
                }
            }
            None => unreachable!("use of dropped segment"),
        }
    }
}

impl<T> Index<usize> for Segment<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index) {
            Some(t) => t,
            None => unreachable!("segment index out of bounds"),
        }
    }
}

impl<T> IndexMut<usize> for Segment<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self.get_mut(index) {
            Some(t) => t,
            None => unreachable!("segment index out of bounds"),
        }
    }
}

impl<T> Drop for Segment<T> {
    fn drop(&mut self) {
        match self.0.take() {
            Some(mut ptr) => unsafe {
                // Drop any remaining elements
                let header = ptr.as_mut();
                let len = header.len();
                header.set_len(0);
                for i in 0..len {
                    header.data().offset(i as isize).drop_in_place()
                }
                // De-allocate the memory associated with the segment
                Header::deallocate(ptr.as_ptr())
            },
            None => {}
        }
    }
}

#[cfg(test)]
mod test_raw {
    use super::*;

    #[test]
    fn test_header() {
        #[repr(C, align(128))]
        struct Funk([u8; 3]);

        let h = Header::<Funk>::allocate(NonZeroUsize::new(1024).unwrap());
        let h_ptr = unsafe { h.as_ref() }.data();
        assert_eq!(mem::size_of::<Funk>(), 128);
        assert_eq!(mem::align_of::<Funk>(), 128);
        assert_eq!(Header::<Funk>::padding(), 112);
        assert_eq!(h_ptr.align_offset(mem::align_of::<Funk>()), 0);
    }

    #[test]
    fn test_segment() {
        let mut s = Segment::<i32>::new(NonZeroUsize::new(64).unwrap());
        assert_eq!(
            std::mem::size_of::<Segment<i32>>(),
            std::mem::size_of::<usize>()
        );

        assert_eq!(0, s.len());
        assert_eq!(64, s.capacity());

        s.push(1);
        s.push(2);
        s.push(3);
        s.push(4);
        s.push(5);
        s.push(6);
        s[0] = s[1];
        s[5] = s[4];
        assert_eq!(6, s.len());

        assert_eq!(s.pop(), Some(5));
        assert_eq!(s.pop(), Some(5));
        assert_eq!(s.pop(), Some(4));
        assert_eq!(s.pop(), Some(3));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.pop(), Some(2));
        assert!(s.pop().is_none());
        assert_eq!(0, s.len());
    }
}
